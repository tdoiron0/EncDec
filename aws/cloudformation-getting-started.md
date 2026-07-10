# Getting Started with AWS CloudFormation

CloudFormation is AWS's native infrastructure-as-code service: you write a declarative template describing the AWS resources you want, hand it to CloudFormation, and it creates, updates, or tears down the real resources to match. If you've used `docker-compose` to declare a set of containers, the mental model is similar — except here you're declaring S3 buckets, IAM roles, EC2 instances, Lambda functions, and the rest of AWS. For an MLOps pipeline, it's how you'd codify the infrastructure your services run on instead of clicking through the console by hand.

## The mental model

Three terms do most of the work:

- **Template** — a YAML (or JSON) file describing your desired resources. This is the artifact you write and version in git.
- **Stack** — a deployed instance of a template. CloudFormation tracks every resource in a stack as a unit, so the whole thing can be updated or deleted together.
- **Reconciliation** — you describe the *end state*, not the steps. When you change a template and redeploy, CloudFormation diffs current reality against the template and figures out what to create, modify, or destroy.

## Prerequisites

You'll need an AWS account and the AWS CLI installed and configured (`aws configure`, with an access key that has permission to create the resources below). Everything here uses the CLI rather than the console, since that's what you'll actually wire into CI later.

## Template anatomy

A template has a handful of top-level sections. Only one is required:

```yaml
AWSTemplateFormatVersion: '2010-09-09'   # optional, only valid value
Description: A human-readable description  # optional

Parameters:    # optional — inputs you pass at deploy time
Mappings:      # optional — static lookup tables (e.g. region -> AMI id)
Conditions:    # optional — toggle resources on/off based on parameters
Resources:     # REQUIRED — the actual AWS resources
Outputs:       # optional — values to return (bucket ARNs, endpoint URLs, etc.)
```

`Resources` is the only section you can't skip. Everything else exists to make your templates reusable and to surface useful values.

## Your first stack — an S3 bucket

S3 is the classic "hello world" because a bucket needs no other infrastructure. Save this as `template.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: My first CloudFormation stack

Resources:
  MyBucket:
    Type: AWS::S3::Bucket
```

That's a complete, valid template. Notice there's no bucket name — if you omit it, CloudFormation generates a globally-unique name for you, which is usually what you want (bucket names must be globally unique and lowercase, so hardcoding them invites collisions).

Deploy it:

```bash
aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name my-first-stack
```

`deploy` is the convenient command: it creates the stack if it doesn't exist and updates it if it does, so you don't have to track which case you're in. Once it finishes, check the AWS console (CloudFormation → Stacks) or run `aws cloudformation describe-stacks --stack-name my-first-stack` and you'll see your bucket, managed by the stack.

## Making it real — parameters, refs, and outputs

Hardcoded templates aren't reusable. Here's the same bucket with a configurable name, versioning turned on, and an output exposing the bucket's ARN:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: S3 bucket with a configurable name and versioning

Parameters:
  BucketName:
    Type: String
    Description: Name for the S3 bucket

Resources:
  DataBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Ref BucketName
      VersioningConfiguration:
        Status: Enabled

Outputs:
  BucketArn:
    Description: ARN of the created bucket
    Value: !GetAtt DataBucket.Arn
```

Deploy it, passing the parameter:

```bash
aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name data-stack \
  --parameter-overrides BucketName=trent-ml-data-12345
```

`!Ref BucketName` pulls in the parameter value. `!GetAtt DataBucket.Arn` reads an attribute off the resource after it's created. The `Outputs` block makes that ARN visible in `describe-stacks` and — importantly — available to *other* stacks that import it, which is how larger setups get composed.

## Intrinsic functions you'll use constantly

These are CloudFormation's built-in functions. The `!` is YAML shorthand for the longer `Fn::` form.

- **`!Ref logicalName`** — returns the value of a parameter, or a resource's default identifier (for an S3 bucket, its name; for an EC2 instance, its instance ID).
- **`!GetAtt resource.Attribute`** — returns a specific attribute of a resource, like `.Arn`, `.DomainName`, or `.PrivateIp`. Each resource type documents which attributes it exposes.
- **`!Sub`** — string interpolation: `!Sub 'arn:aws:s3:::${BucketName}/*'` substitutes parameters and resource references inline.
- **`!Join`** — joins a list with a delimiter: `!Join ['-', ['my', 'app', !Ref Env]]`.
- **`!FindInMap`** — looks a value up in a `Mappings` table, commonly used to pick a region-specific AMI.

`Ref`, `GetAtt`, and `Sub` cover the large majority of real templates.

## Updating a stack

Edit the template and run the same `deploy` command. CloudFormation computes the diff and applies only what changed. If you want to *preview* changes before they happen — which you'll want for anything important — use a change set:

```bash
aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name data-stack \
  --parameter-overrides BucketName=trent-ml-data-12345 \
  --no-execute-changeset
```

That prints a change set you can inspect with `aws cloudformation describe-change-set`, then execute deliberately. One quirk: if there are no changes, `deploy` exits with an error by default. Add `--no-fail-on-empty-changeset` in scripts so a no-op doesn't fail your pipeline.

## Cleaning up

This is the part to internalize early, since leftover resources cost money:

```bash
aws cloudformation delete-stack --stack-name data-stack
```

CloudFormation deletes every resource in the stack. Because the stack tracks everything as a unit, you don't have to hunt down individual resources — one of the real advantages over clicking things into existence manually. (Note: a non-empty S3 bucket won't delete unless you've emptied it or configured it accordingly, so that one occasionally needs a manual emptying first.)

## A gotcha worth knowing early — capabilities

The moment your template creates IAM resources (roles, policies — which most real workloads need), `deploy` will refuse unless you explicitly acknowledge it:

```bash
aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name my-stack \
  --capabilities CAPABILITY_IAM
```

Use `CAPABILITY_NAMED_IAM` instead if you give your IAM resources explicit names. This trips up almost everyone on their first IAM template, so it's good to recognize the error before it puzzles you.

## Where to go next

- **Validate before deploying** — `aws cloudformation validate-template` catches syntax errors, and `cfn-lint` (a separate, widely-used linter) catches a lot more, including invalid resource properties.
- **AWS SAM** — the Serverless Application Model is a thin extension of CloudFormation with much terser syntax for Lambda, API Gateway, and DynamoDB. Worth it the moment you deploy a serverless inference endpoint.
- **Nested and cross-stack references** — break large infrastructure into composable stacks and wire them together via exported outputs.
- **CI integration** — since you already use GitHub Actions, a natural next step is a workflow that runs `cfn-lint`, then `aws cloudformation deploy` on merge to main. That's the IaC-in-CI story that reads well on an MLOps resume and directly addresses the "major cloud platforms" gap.

Start by deploying the first S3 example end to end, then delete it, so you've seen the full create-and-destroy loop before adding complexity.
